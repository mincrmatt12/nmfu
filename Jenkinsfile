pipeline {
	agent {
		dockerfile {
			filename 'Dockerfile.build'
		}
	}
	stages {
		stage ('Build') {
			steps {
				sh "python setup.py egg_info --tag-build 'dev0+git-${env.GIT_COMMIT}' sdist bdist_wheel"
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