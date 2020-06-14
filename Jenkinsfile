pipeline {
	agent {
		dockerfile {
			filename 'Dockerfile.build'
		}
	}
	stages {
		stage ('Build') {
			environment {
				TAG_BUILD = """${sh(returnStdout: true, script: 'bash -c "[[ ${TAG_NAME} ]] && true || echo dev0+git-${GIT_COMMIT}"').trim()}"""
			}
			steps {
				sh "python setup.py egg_info --tag-build ${TAG_BUILD} sdist bdist_wheel"
				archiveArtifacts artifacts: 'dist/*'
			}
		}
		stage ('Test') {
			steps {
				sh "pytest --junit-xml=junit.xml || true"
				junit 'junit.xml'
			}
		}
	}
}
