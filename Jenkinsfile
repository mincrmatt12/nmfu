pipeline {
	agent {
		docker { image 'python:3' }
	}
	stages {
		stage ('Build') {
			steps {
				sh "python setup.py egg_info --tag-build 'dev0+git-${env.GIT_COMMIT}' sdist bdist_wheel"
				archiveArtifacts artifacts: 'dist/*'
			}
		}

		stage ('Test') {
			agent {
				dockerfile {
					filename 'Dockerfile.test'
				}
			}
			steps {
				sh "pip install .[tests]"
				sh "pytest --junit-xml=junit.xml"
				junit 'junit.xml'
			}
		}
	}
}
